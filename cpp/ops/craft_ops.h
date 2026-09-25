#pragma once

#include <cstdint>
#include <vector>

/**
 * CRAFT heatmap decoding: the detector-specific kernel that
 * react-native-executorch dropped when it scoped its OCR down to DBNet.
 * JSI-free so it can be unit-tested on the host; `bindings.cpp` exposes it to
 * JavaScript.
 */
namespace rnocr::ops::craft {

/**
 * One connected text component: an axis-aligned box (`x0, y0` = min,
 * `x1, y1` = max) in the caller's coordinate space plus the angle of the
 * rotated rectangle fitted to the component, in degrees. Line grouping and
 * de-skew happen in TypeScript.
 */
struct CraftBox {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float angle = 0.0f;
};

struct CraftOptions {
    /** Region-score threshold for a pixel to count as text. */
    float textThreshold = 0.4f;
    /** Affinity-score threshold for linking adjacent glyphs. */
    float linkThreshold = 0.4f;
    /** Minimum peak region score for a component to survive. */
    float lowTextThreshold = 0.7f;
    /** Components covering fewer heatmap pixels than this are dropped. */
    int32_t minArea = 10;
    /**
     * `false` adds the affinity map so linked glyphs form one line-level
     * component (angle kept); `true` subtracts it, then erodes and dilates, so
     * every glyph is its own upright component (angle 0).
     */
    bool charLevel = false;
    /**
     * Multiplier from heatmap pixels to the returned coordinates. CRAFT's
     * heatmap is half the detector input, so callers pass
     * `inputHeight / heatHeight`. Must be positive.
     */
    float scale = 1.0f;
};

/**
 * Decodes a CRAFT heatmap into component boxes.
 *
 * @param heat `float32 [heatHeight, heatWidth, 2]`, the text and affinity
 * scores interleaved per pixel, as the model emits them.
 * @returns One box per surviving component, in no particular order, with
 * coordinates inside the heatmap bounds and multiplied by `options.scale`.
 * @throws core::error::RnOcrException with code InvalidArgument for empty
 * dimensions, a negative `minArea` or a non-positive `scale`; ExecutionFailed
 * when OpenCV fails.
 */
std::vector<CraftBox> extractCraftTextBoxes(const float *heat, int32_t heatHeight, int32_t heatWidth, const CraftOptions &options);

} // namespace rnocr::ops::craft
