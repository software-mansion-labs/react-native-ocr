# Contributing

Contributions are always welcome, no matter how large or small!

We want this community to be friendly and respectful to each other. Please follow it in all your
interactions with the project. Before contributing, please read the
[code of conduct](./CODE_OF_CONDUCT.md).

## Development workflow

This project is a monorepo managed using [Yarn workspaces](https://yarnpkg.com/features/workspaces).
It contains the following packages:

- The library package in the root directory.
- An example app in the `example/` directory.

To get started with the project, run `yarn` in the root directory to install the required
dependencies for each package:

```sh
yarn
```

> Since the project relies on Yarn workspaces, you cannot use [`npm`](https://github.com/npm/cli)
> for development.

The install also runs react-native-executorch's postinstall, which downloads its native ExecuTorch
libraries from a GitHub release pinned by `nativeLibsVersion` in its `package.json`
(`0.10.6-libs` since react-native-executorch 0.10.6, the first release whose Vulkan backend carries
the Adreno pipeline-creation fix). Apps get the libraries their own react-native-executorch version
pins, which is why the peer dependency starts at 0.10.6.

The [example app](/example/) demonstrates usage of the library. You need to run it to test any
changes you make.

It is configured to use the local version of the library, so any changes you make to the library's
source code will be reflected in the example app. Changes to the library's JavaScript code will be
reflected in the example app without a rebuild, but native code changes will require a rebuild of
the example app. The example is an Expo project: `expo run:*` generates `example/ios` and
`example/android`, which are git-ignored. Yarn creates no `node_modules` link for the root
workspace, so `example/react-native.config.js` points autolinking at the repository root
explicitly and Metro resolves the library through the Babel alias `react-native-builder-bob` adds
in `example/babel.config.js`.

You can use various commands from the root directory to work with the project.

To start the packager:

```sh
yarn example start
```

If another Metro already holds port 8081, pass `--port` to the command.

To run the example app on Android:

```sh
yarn example android
```

To run the example app on iOS:

```sh
yarn example ios
```

To confirm that the app is running with the new architecture, you can check the Metro logs for a
message like this:

```sh
Running "main" with {"fabric":true,"initialProps":{},"rootTag":1}
```

Note the `"fabric":true` property.

Make sure your code passes TypeScript and ESLint. Run the following to verify:

```sh
yarn typecheck
yarn lint
```

To fix formatting errors, run the following:

```sh
yarn lint --fix
yarn format
```

Remember to add tests for your change if possible. Run the unit tests by:

```sh
yarn test
```

### Repository layout

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

### Native code

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
- After adding or removing a file under `cpp/`, run `pod install` in `example/ios` before
  `yarn example ios`; the Xcode project lists files explicitly. Android's CMake globs the tree.

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
seconds rather than in an Xcode or Gradle build.

### Sample pages

The pages under `example/assets/ocr/` and the text they carry (`expected.json`) come from
`scripts/make-ocr-sample.py` (Pillow with raqm, NumPy, the macOS system fonts). Edit the script
rather than the images; it checks every character of the per-script pages against the matching
alphabet in `src/charsets.ts`, so a line the model cannot read in principle never ends up as an
expected line.

### Scripted runs

The EasyOCR and Document screens open in a given state from a deep link
(`rnocr-example://easyocr?lang=KOREAN&backend=COREML&sample=vertical`,
`rnocr-example://document?sample=table&backend=COREML&helpersBackend=COREML&layout=1&tables=1`)
and run as soon as the models are ready, so a page can be measured without tapping. Two
parameters exist for trying exports before they are published: `runs=N` runs the page N times
back to back and lists every time in `Documents/last-run.json` (which already holds the last
run's reads and grade), and `modelUrl`, plus `helpersUrl` and `layoutUrl` on the Document
screen, swap a registry export for another file. The value is a URL, or a bare file name for a
`.pte` pushed into the app's documents directory (`adb push`, or `xcrun devicectl device copy
to --domain-type appDataContainer --domain-identifier <bundle id> --destination Documents/<file>`
on an iPhone). `scripts/measure-page.sh <device> '<screen>?<query>' [runs]` drives the whole
loop from the Mac: it relaunches the app, opens the link, waits for the runs and prints the warm
median with the files the run used, for an iPhone UDID, the booted simulator or an adb serial
(`APP_ID` overrides the bundle id).

### Docs

`yarn docs` builds the API reference with typedoc into `docs/` (git-ignored). The `Docs` workflow
builds it on every push to `main` and deploys it to GitHub Pages; the repository's Pages source
must be set to "GitHub Actions" once.

### Commit message convention

We follow the [conventional commits specification](https://www.conventionalcommits.org/en) for our
commit messages:

- `fix`: bug fixes, e.g. fix crash due to deprecated method.
- `feat`: new features, e.g. add new method to the module.
- `refactor`: code refactor, e.g. migrate from class components to hooks.
- `docs`: changes into documentation, e.g. add usage example for the module.
- `test`: adding or updating tests, e.g. add integration tests using detox.
- `chore`: tooling changes, e.g. change CI config.

Our pre-commit hooks verify that your commit message matches this format when committing.

### Linting and tests

[ESLint](https://eslint.org/), [Prettier](https://prettier.io/),
[TypeScript](https://www.typescriptlang.org/)

We use [TypeScript](https://www.typescriptlang.org/) for type checking,
[ESLint](https://eslint.org/) with [Prettier](https://prettier.io/) for linting and formatting the
code, [Jest](https://jestjs.io/) for testing the TypeScript and
[GoogleTest](https://github.com/google/googletest) for testing the C++ kernels.

Our pre-commit hooks verify that the linter and tests pass when committing; `clang-format` formats
C++ when installed (`brew install clang-format`), and CI checks it.

### Publishing to npm

Releases are published from CI with the `Publish to npm` workflow (`workflow_dispatch`), which
runs Software Mansion's shared `npm-package-publish` action: pick the release type, give the
version explicitly, and leave `dry-run` on for a rehearsal. A stable release without an explicit
version infers it from a `x.y-stable` branch name, so a run from `main` fails without one. The
action resolves the version, runs `npm pack` (which runs `bob build` through `prepare`), uploads
the tarball as a workflow artifact, publishes with npm provenance, and with
`perform-git-operations` on pushes the `vX.Y.Z` tag (the release commit is skipped on `main`, so
a protected `main` does not block it). The package has to allow the workflow as a trusted
publisher on npm; that setting lives in the package's settings on npmjs.com, so the very first
version is published by hand (see release-it below) by someone with publish rights in the
`@swmansion` organisation, and the trusted publisher is added right after. Before a release:

```sh
yarn typecheck && yarn lint && yarn test && yarn test:native
npm pack --dry-run      # check the file list: src, lib, android, ios, cpp (without tests), podspec
```

We also keep [release-it](https://github.com/release-it/release-it) as a local fallback for
someone with publish rights. It handles common tasks like bumping version based on semver,
creating tags and releases etc.:

```sh
yarn release                  # or: yarn release --no-increment, when package.json already holds the version
```

### Scripts

The `package.json` file contains various scripts for common tasks:

- `yarn`: setup project by installing dependencies.
- `yarn typecheck`: type-check files with TypeScript.
- `yarn lint`: lint files with ESLint.
- `yarn format`: format files with Prettier.
- `yarn test`: run unit tests with Jest.
- `yarn test:native`: run the GoogleTest suites for the C++ kernels.
- `yarn prepare`: build the library into `lib/` with react-native-builder-bob.
- `yarn docs`: build the API reference with typedoc.
- `yarn example start`: start the Metro server for the example app.
- `yarn example android`: run the example app on Android.
- `yarn example ios`: run the example app on iOS.

### Sending a pull request

> **Working on your first pull request?** You can learn how from this _free_ series:
> [How to Contribute to an Open Source Project on GitHub](https://app.egghead.io/playlists/how-to-contribute-to-an-open-source-project-on-github).

When you're sending a pull request:

- Prefer small pull requests focused on one change.
- Verify that linters and tests are passing.
- Review the documentation to make sure it looks good.
- For pull requests that change the API or implementation, discuss with maintainers first by
  opening an issue.
