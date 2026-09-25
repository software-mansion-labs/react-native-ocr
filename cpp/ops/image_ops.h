#pragma once

#include <cstdint>

/**
 * Whole-image kernels: the page-level operations the OCR pipeline needs that
 * react-native-executorch does not expose. Every kernel is a plain C++
 * function over raw `uint8` `[height, width, channels]` buffers, JSI-free so
 * it can be unit-tested on the host; `bindings.cpp` exposes each one to
 * JavaScript. Callers size the output buffers; the kernels never allocate
 * them.
 *
 * Every kernel throws core::error::RnOcrException: InvalidArgument for a
 * contract violation, ExecutionFailed when OpenCV itself fails.
 */
namespace rnocr::ops::image {

/**
 * Rotates an image clockwise by 90, 180 or 270 degrees into `dst`, which must
 * hold `[width, height, channels]` for 90 and 270 and
 * `[height, width, channels]` for 180.
 */
void rotate(const uint8_t *src, int32_t height, int32_t width, int32_t channels, int32_t degreesClockwise, uint8_t *dst);

/**
 * Copies the axis-aligned region `[x0, x1) x [y0, y1)` of an image into `dst`,
 * which must hold `[y1 - y0, x1 - x0, channels]`. The bounds must satisfy
 * `0 <= x0 < x1 <= width` and `0 <= y0 < y1 <= height`; clamping is the
 * caller's job so the output size is known up front.
 */
void crop(const uint8_t *src, int32_t height, int32_t width, int32_t channels, int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint8_t *dst);

/** How `warpByGrid` samples between source pixels. */
enum class Interpolation { Nearest,
                           Linear,
                           Cubic };

/** How `warpByGrid` fills samples that fall outside the source. */
enum class BorderMode { Replicate,
                        Constant };

struct WarpOptions {
    Interpolation interpolation = Interpolation::Linear;
    BorderMode borderMode = BorderMode::Replicate;
    /** Fill value for every channel under BorderMode::Constant. */
    double borderValue = 0.0;
};

/**
 * Resamples an image through a `grid_sample`-style backward field into `dst`,
 * which must hold `[dstHeight, dstWidth, channels]`.
 *
 * `grid` is `float32 [2, gridHeight, gridWidth]`: channel 0 holds x and
 * channel 1 holds y, both normalized to `[-1, 1]` with `align_corners = true`
 * against the source dimensions, exactly as UVDoc emits it. The grid is
 * bilinearly upsampled to the destination size first, so it may be far coarser
 * than the image, and the destination may differ in size from the source.
 * Sampling is `cv::remap`. Images of 1 to 4 channels are supported.
 */
void warpByGrid(const uint8_t *src, int32_t srcHeight, int32_t srcWidth, int32_t channels,
                const float *grid, int32_t gridHeight, int32_t gridWidth,
                int32_t dstHeight, int32_t dstWidth, const WarpOptions &options, uint8_t *dst);

/**
 * Widens a recognizer strip to `targetWidth` into `dst`, which must hold
 * `[height, targetWidth, channels]`: the strip is copied left-aligned and the
 * new columns take the mean color of its four corner patches, so the padding
 * continues the background instead of introducing a seam some recognizers read
 * as a glyph. `targetWidth >= width`; images of 1 to 4 channels are supported.
 */
void padCornerMean(const uint8_t *strip, int32_t height, int32_t width, int32_t channels, int32_t targetWidth, uint8_t *dst);

} // namespace rnocr::ops::image
