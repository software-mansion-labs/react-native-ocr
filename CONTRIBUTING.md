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

react-native-executorch downloads its native ExecuTorch libraries from a GitHub release pinned by
`nativeLibsVersion` in its `package.json`. This repository pins that to `0.10.6-libs` (the
Vulkan backend rebuilt with the Adreno pipeline-creation fix) on top of the published
`react-native-executorch@0.10.5`, which still points at `0.10.4-libs`, through a Yarn patch
(`.yarn/patches/`, wired in `resolutions`). Drop the patch once an npm release of
react-native-executorch pins `0.10.6` or later itself. The published package does not carry the
pin; apps get the libraries their own react-native-executorch version pins.

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
yarn test:native      # GoogleTest suites for the C++ kernels (see below)
```

Commits follow the Conventional Commits format (`feat:`, `fix:`, `chore:`, ...). Lefthook runs
lint, typecheck, tests and formatting on staged files; `clang-format` formats C++ when installed
(`brew install clang-format`).

## Native code

- OpenCV comes from react-native-executorch: the `opencv-rne` CocoaPod on iOS and the static
  libraries its postinstall downloads on Android. The app must declare the `ocr` feature.
- The kernels in `cpp/ops/{image_ops,craft_ops}.cpp` are plain C++ over raw buffers and never
  include JSI; `cpp/ops/bindings.cpp` wraps each one as a host function. Keep new kernels on that
  side of the line so the host tests can build them.
- Every JSI host function takes and returns plain `ArrayBuffer`s with explicit shapes; the
  TypeScript wrappers in `src/native/ops.ts` bridge them to react-native-executorch tensors, and
  `src/native/types.ts` mirrors the native signatures by hand.
- Errors thrown from C++ reach JavaScript as `RnExecuTorchError`s with a `code`, so callers see one
  error type across both packages. Kernels throw `core::error::RnOcrException`; `core/guard.h`
  converts it at the boundary. Wrap OpenCV calls in `ops::withCv`, and never write
  `catch (const cv::Exception &)`: the prebuilt mobile OpenCV libraries have no RTTI, so that
  catch needs a typeinfo they do not export and the app fails to link (the host build, which has
  RTTI, would not notice).
- Every op has a JavaScript fake in `__tests__/support/fakeRnOcr.ts` so the Jest suites run the
  TypeScript wrappers end to end. Add one whenever you add an op.

### C++ tests

`cpp/tests/` holds GoogleTest suites for the kernels. They build against the OpenCV installed on
the machine and run in a second or two:

```sh
brew install opencv googletest          # macOS; GoogleTest is fetched if missing
yarn test:native                        # or scripts/run-native-tests.sh -R CraftOpsTest
```

On Debian/Ubuntu install `libopencv-core-dev libopencv-imgproc-dev` instead. The device builds
link OpenCV 4.x from react-native-executorch while Homebrew ships 5.x; the kernels compile against
both, and `craft_ops.cpp` shows the one include that differs.

When `node_modules` is present the same build also compiles the JSI layer (`cpp/ops/bindings.cpp`,
`cpp/core/*`) against react-native's `jsi.h` without linking it, so a type error there shows up in
seconds rather than in an Xcode or Gradle build. After adding or removing a file under `cpp/`, run
`pod install` in `example/ios` before `yarn example ios`; the Xcode project lists files explicitly.

## Sample pages

The pages under `example/assets/ocr/` and the text they carry (`expected.json`) come from
`scripts/make-ocr-sample.py` (Pillow with raqm, NumPy, the macOS system fonts). Edit the script
rather than the images; it checks every character of the per-script pages against the matching
alphabet in `src/charsets.ts`, so a line the model cannot read in principle never ends up as an
expected line.

## Docs

`yarn docs` builds the API reference with typedoc into `docs/` (git-ignored). The `Docs` workflow
builds it on every push to `main` and deploys it to GitHub Pages; the repository's Pages source
must be set to "GitHub Actions" once.

## Releasing

Releases are published from CI with the `Publish to npm` workflow (`workflow_dispatch`), which
runs Software Mansion's shared `npm-package-publish` action: pick the release type, optionally an
explicit version, and leave `dry-run` on for a rehearsal. The action resolves the version, runs
`npm pack` (which runs `bob build` through `prepare`), uploads the tarball as a workflow artifact,
publishes with npm provenance, and with `perform-git-operations` on commits the version and pushes
the `vX.Y.Z` tag. The package has to allow the workflow as a trusted publisher on npm. Before a
release:

```sh
yarn typecheck && yarn lint && yarn test && yarn test:native
npm pack --dry-run      # check the file list: src, lib, android, ios, cpp (without tests), podspec
```

`yarn release` (release-it) remains as a local fallback for someone with publish rights; use
`--no-increment` when `package.json` already holds the version to publish.
