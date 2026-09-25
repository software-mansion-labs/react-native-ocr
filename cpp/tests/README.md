# C++ unit tests

Host-side GoogleTest suites for the kernels under `cpp/ops/`. They run on a developer machine or
a CI runner, with no simulator, emulator or JavaScript engine involved.

The kernels are plain C++ functions over raw buffers (`image_ops.h`, `craft_ops.h`) and never
include JSI, so this build compiles just those two files against the OpenCV installed on the
host. The JSI bindings (`cpp/ops/bindings.cpp`, `cpp/core/*`) are compile-checked here against
react-native's `jsi.h` when `node_modules` is present (nothing links them), and exercised by the
example app's spike screen and by the Jest suites, which run the TypeScript wrappers against a
JavaScript fake of the same contract.

The host OpenCV has RTTI and the prebuilt mobile ones do not, so a `catch (const cv::Exception &)`
passes here and fails to link on device; always catch through `std::exception` (`ops::withCv`).

## Running

```sh
brew install opencv googletest        # macOS
apt-get install libopencv-core-dev libopencv-imgproc-dev   # Debian/Ubuntu (GoogleTest is fetched)

scripts/run-native-tests.sh           # configure + build + ctest, from anywhere
scripts/run-native-tests.sh -R CraftOpsTest   # extra arguments go to ctest
```

The build tree is `cpp/tests/build/` (git-ignored). Homebrew currently ships OpenCV 5, whose
`geometry` module holds `minAreaRect`; the CMake project links it when present and the device
builds (OpenCV 4.x from react-native-executorch) keep using `imgproc`.

## Layout

| Path                    | Contents                                                             |
| ----------------------- | -------------------------------------------------------------------- |
| `ImageOpsTest.cpp`      | `rotate`, `crop`, `warpByGrid`, `padCornerMean`                     |
| `CraftOpsTest.cpp`      | `extractCraftTextBoxes` on synthetic heatmaps                        |
| `support/Assertions.h`  | `expectThrowsCode`: asserts a kernel raises a coded `RnOcrException` |

One binary per suite, so `ctest -R ImageOpsTest` targets one and a crash in one area cannot take
the rest down.
