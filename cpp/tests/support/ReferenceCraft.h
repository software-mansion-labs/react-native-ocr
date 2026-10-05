#pragma once

#include <cstdint>
#include <vector>

/**
 * The CRAFT heatmap decoder of react-native-executorch at commit
 * 3f83e48a18bdb5c73ff5d65d364129b29e9a1578
 * (`packages/react-native-executorch/cpp/extensions/cv/ocr_ops.cpp`), the
 * implementation `cpp/ops/craft_ops.cpp` was ported from. It exists only so
 * `CraftParityTest` can run both decoders over the same real heatmaps; nothing
 * in the package links it.
 */
namespace rnocr::tests::reference {

struct Box {
    float x0{}, y0{}, x1{}, y1{};
    float angle = 0.0f;
};

/**
 * `extractCraft` of the reference: `data` points at `heatW * heatH * 2` floats
 * (text and affinity interleaved); the boxes come back in heatmap pixels times
 * `restoreRatio`.
 */
std::vector<Box> extractCraft(float *data, int32_t heatW, int32_t heatH, float textThreshold, float linkThreshold,
                              float lowTextThreshold, float restoreRatio, bool charLevel);

} // namespace rnocr::tests::reference
