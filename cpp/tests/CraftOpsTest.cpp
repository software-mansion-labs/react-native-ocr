#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "ops/craft_ops.h"
#include "support/Assertions.h"

namespace rnocr::tests {
namespace {

using core::error::ErrorCode;
using ops::craft::CraftBox;
using ops::craft::CraftOptions;
using ops::craft::extractCraftTextBoxes;

/** A synthetic `[h, w, 2]` heatmap with rectangular blobs painted into it. */
struct Heat {
    int32_t height;
    int32_t width;
    std::vector<float> data;

    Heat(int32_t h, int32_t w) : height(h), width(w), data(static_cast<size_t>(h) * w * 2, 0.0f) {}

    /** Paints `value` into the text channel over the inclusive rectangle. */
    Heat &text(int x0, int y0, int x1, int y1, float value = 1.0f) {
        return paint(0, x0, y0, x1, y1, value);
    }

    /** Paints `value` into the affinity channel over the inclusive rectangle. */
    Heat &affinity(int x0, int y0, int x1, int y1, float value = 1.0f) {
        return paint(1, x0, y0, x1, y1, value);
    }

    std::vector<CraftBox> decode(const CraftOptions &options = CraftOptions{}) const {
        return extractCraftTextBoxes(data.data(), height, width, options);
    }

private:
    Heat &paint(int channel, int x0, int y0, int x1, int y1, float value) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                data[(static_cast<size_t>(y) * width + x) * 2 + channel] = value;
            }
        }
        return *this;
    }
};

CraftOptions withCharLevel(bool charLevel) {
    CraftOptions options;
    options.charLevel = charLevel;
    return options;
}

/** True when `angle` is a multiple of 90 degrees within a tolerance. */
bool isAxisAligned(float angle) {
    const float remainder = std::fmod(std::fabs(angle), 90.0f);
    return remainder < 0.5f || remainder > 89.5f;
}

TEST(CraftOpsTest, ReturnsOneBoxForASolidBlob) {
    const auto boxes = Heat(32, 32).text(8, 8, 15, 15).decode();
    ASSERT_EQ(boxes.size(), 1u);
    const CraftBox &box = boxes[0];
    // The box covers the blob and, after dilation by the stroke-width estimate,
    // does not run away from it.
    EXPECT_LE(box.x0, 8.0f);
    EXPECT_LE(box.y0, 8.0f);
    EXPECT_GE(box.x1, 15.0f);
    EXPECT_GE(box.y1, 15.0f);
    EXPECT_GE(box.x0, 0.0f);
    EXPECT_GE(box.y0, 0.0f);
    EXPECT_LE(box.x1, 24.0f);
    EXPECT_LE(box.y1, 24.0f);
    EXPECT_TRUE(isAxisAligned(box.angle)) << "angle " << box.angle;
}

TEST(CraftOpsTest, ReturnsNothingForAnEmptyMap) {
    EXPECT_TRUE(Heat(16, 16).decode().empty());
}

TEST(CraftOpsTest, FindsEveryDisjointBlob) {
    const auto boxes = Heat(48, 48).text(4, 4, 15, 15).text(30, 30, 43, 43).decode();
    EXPECT_EQ(boxes.size(), 2u);
}

TEST(CraftOpsTest, ScaleMultipliesTheCoordinates) {
    const Heat heat = Heat(32, 32).text(8, 8, 15, 15);
    const auto unit = heat.decode();
    CraftOptions options;
    options.scale = 2.0f;
    const auto doubled = heat.decode(options);
    ASSERT_EQ(unit.size(), 1u);
    ASSERT_EQ(doubled.size(), 1u);
    EXPECT_FLOAT_EQ(doubled[0].x0, unit[0].x0 * 2.0f);
    EXPECT_FLOAT_EQ(doubled[0].y0, unit[0].y0 * 2.0f);
    EXPECT_FLOAT_EQ(doubled[0].x1, unit[0].x1 * 2.0f);
    EXPECT_FLOAT_EQ(doubled[0].y1, unit[0].y1 * 2.0f);
    EXPECT_FLOAT_EQ(doubled[0].angle, unit[0].angle);
}

// Two glyph blobs joined by an affinity region: the line-level decode links
// them into one component, the char-level decode keeps them apart.
Heat linkedPair() {
    return Heat(40, 40).text(4, 8, 11, 15).text(20, 8, 27, 15).affinity(12, 8, 19, 15);
}

TEST(CraftOpsTest, AffinityLinksGlyphsIntoOneLineBox) {
    const auto boxes = linkedPair().decode(withCharLevel(false));
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_LE(boxes[0].x0, 4.0f);
    EXPECT_GE(boxes[0].x1, 27.0f);
}

TEST(CraftOpsTest, CharLevelSeparatesLinkedGlyphs) {
    auto boxes = linkedPair().decode(withCharLevel(true));
    ASSERT_EQ(boxes.size(), 2u);
    std::sort(boxes.begin(), boxes.end(), [](const CraftBox &a, const CraftBox &b) { return a.x0 < b.x0; });
    EXPECT_LE(boxes[0].x0, 4.0f);
    EXPECT_GE(boxes[0].x1, 11.0f);
    EXPECT_LE(boxes[1].x0, 20.0f);
    EXPECT_GE(boxes[1].x1, 27.0f);
    EXPECT_LT(boxes[0].x0, boxes[1].x0);
    // Glyphs are read upright.
    EXPECT_FLOAT_EQ(boxes[0].angle, 0.0f);
    EXPECT_FLOAT_EQ(boxes[1].angle, 0.0f);
}

TEST(CraftOpsTest, WithoutAffinityTheGlyphsStaySeparateAtLineLevel) {
    const auto boxes = Heat(40, 40).text(4, 8, 11, 15).text(20, 8, 27, 15).decode(withCharLevel(false));
    EXPECT_EQ(boxes.size(), 2u);
}

TEST(CraftOpsTest, DropsBlobsBelowTheTextThreshold) {
    CraftOptions options;
    options.textThreshold = 0.4f;
    options.lowTextThreshold = 0.2f;
    EXPECT_TRUE(Heat(32, 32).text(8, 8, 15, 15, 0.3f).decode(options).empty());
    EXPECT_EQ(Heat(32, 32).text(8, 8, 15, 15, 0.5f).decode(options).size(), 1u);
}

TEST(CraftOpsTest, DropsComponentsWhosePeakIsBelowLowTextThreshold) {
    CraftOptions options;
    options.textThreshold = 0.4f;
    options.lowTextThreshold = 0.7f;
    // Binarizes (0.5 > 0.4) but never peaks above 0.7.
    EXPECT_TRUE(Heat(32, 32).text(8, 8, 15, 15, 0.5f).decode(options).empty());
    options.lowTextThreshold = 0.5f;
    EXPECT_EQ(Heat(32, 32).text(8, 8, 15, 15, 0.5f).decode(options).size(), 1u);
}

TEST(CraftOpsTest, DropsComponentsSmallerThanMinArea) {
    const Heat heat = Heat(32, 32).text(8, 8, 10, 10); // 3x3 = 9 px
    CraftOptions options;
    options.minArea = 10;
    EXPECT_TRUE(heat.decode(options).empty());
    options.minArea = 9;
    EXPECT_EQ(heat.decode(options).size(), 1u);
    options.minArea = 0;
    EXPECT_EQ(heat.decode(options).size(), 1u);
}

TEST(CraftOpsTest, KeepsBoxesInsideTheMapForBlobsOnTheBorder) {
    const auto boxes = Heat(32, 32).text(0, 0, 7, 7).decode();
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_GE(boxes[0].x0, 0.0f);
    EXPECT_GE(boxes[0].y0, 0.0f);
    EXPECT_LE(boxes[0].x1, 32.0f);
    EXPECT_LE(boxes[0].y1, 32.0f);
}

TEST(CraftOpsTest, RejectsInvalidOptions) {
    const Heat heat = Heat(16, 16).text(4, 4, 11, 11);
    CraftOptions zeroScale;
    zeroScale.scale = 0.0f;
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { heat.decode(zeroScale); });
    CraftOptions negativeArea;
    negativeArea.minArea = -1;
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { heat.decode(negativeArea); });
}

TEST(CraftOpsTest, RejectsEmptyMaps) {
    const Heat heat = Heat(16, 16);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { extractCraftTextBoxes(heat.data.data(), 0, 16, CraftOptions{}); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { extractCraftTextBoxes(heat.data.data(), 16, 0, CraftOptions{}); });
}

} // namespace
} // namespace rnocr::tests
