#pragma once

#include <cstdint>

#include <jsi/jsi.h>

/**
 * Whole-image kernels: the page-level operations the OCR pipeline needs that
 * react-native-executorch does not expose. Each kernel is a plain C++ function
 * over raw buffers (unit-tested on the host) with a thin JSI installer next to
 * it.
 */
namespace rnocr::ops::image {

/**
 * Rotates a `[height, width, channels]` uint8 image clockwise by 90, 180 or
 * 270 degrees into `dst`, which must hold `[width, height, channels]` for 90
 * and 270 and `[height, width, channels]` for 180.
 * @throws core::error::RnOcrException with code InvalidArgument for any other
 * angle or an unsupported channel count.
 */
void rotate(const uint8_t *src, int32_t height, int32_t width, int32_t channels, int32_t degreesClockwise, uint8_t *dst);

/** Installs `rotate(src, height, width, channels, degreesClockwise) -> ArrayBuffer`. */
void install_rotate(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);

} // namespace rnocr::ops::image
