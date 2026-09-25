#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "ops/image_ops.h"
#include "support/Assertions.h"

namespace rnocr::tests {
namespace {

using core::error::ErrorCode;
using namespace ops::image;

/** An `[h, w, ch]` image whose byte value is its row-major index (mod 256). */
std::vector<uint8_t> ramp(int32_t h, int32_t w, int32_t ch = 1) {
    std::vector<uint8_t> out(static_cast<size_t>(h) * w * ch);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<uint8_t>(i % 256);
    }
    return out;
}

std::vector<uint8_t> bytes(std::initializer_list<int> values) {
    std::vector<uint8_t> out;
    for (int v : values) {
        out.push_back(static_cast<uint8_t>(v));
    }
    return out;
}

void expectClose(const std::vector<uint8_t> &actual, const std::vector<uint8_t> &expected, int tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_LE(std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])), tolerance) << "at index " << i;
    }
}

// ---------------------------------------------------------------- rotate ---

TEST(RotateTest, Rotates90ClockwiseAndSwapsAxes) {
    const auto src = ramp(2, 3); // [[0,1,2],[3,4,5]]
    std::vector<uint8_t> dst(6);
    rotate(src.data(), 2, 3, 1, 90, dst.data());
    // The first row becomes the last column.
    EXPECT_EQ(dst, bytes({3, 0, 4, 1, 5, 2}));
}

TEST(RotateTest, Rotates180AndKeepsAxes) {
    const auto src = ramp(2, 3);
    std::vector<uint8_t> dst(6);
    rotate(src.data(), 2, 3, 1, 180, dst.data());
    EXPECT_EQ(dst, bytes({5, 4, 3, 2, 1, 0}));
}

TEST(RotateTest, Rotates270Clockwise) {
    const auto src = ramp(2, 3);
    std::vector<uint8_t> dst(6);
    rotate(src.data(), 2, 3, 1, 270, dst.data());
    EXPECT_EQ(dst, bytes({2, 5, 1, 4, 0, 3}));
}

TEST(RotateTest, CarriesEveryChannel) {
    const auto src = bytes({1, 2, 3, 4, 5, 6}); // [1, 2, 3]
    std::vector<uint8_t> dst(6);
    rotate(src.data(), 1, 2, 3, 90, dst.data()); // -> [2, 1, 3]
    EXPECT_EQ(dst, bytes({1, 2, 3, 4, 5, 6}));
}

TEST(RotateTest, RejectsOtherAngles) {
    const auto src = ramp(2, 3);
    std::vector<uint8_t> dst(6);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { rotate(src.data(), 2, 3, 1, 45, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { rotate(src.data(), 2, 3, 1, 0, dst.data()); });
}

TEST(RotateTest, RejectsEmptyImages) {
    std::vector<uint8_t> dst(6);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { rotate(dst.data(), 0, 3, 1, 90, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { rotate(dst.data(), 2, 3, 0, 90, dst.data()); });
}

// ------------------------------------------------------------------ crop ---

TEST(CropTest, CopiesTheRegion) {
    // 3x4: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    const auto src = ramp(3, 4);
    std::vector<uint8_t> dst(2 * 2);
    crop(src.data(), 3, 4, 1, 1, 1, 3, 3, dst.data());
    EXPECT_EQ(dst, bytes({5, 6, 9, 10}));
}

TEST(CropTest, CopiesEveryChannelOfTheRegion) {
    const auto src = ramp(2, 2, 3); // pixel (y, x) = [6y+3x, +1, +2]
    std::vector<uint8_t> dst(3);
    crop(src.data(), 2, 2, 3, 1, 1, 2, 2, dst.data());
    EXPECT_EQ(dst, bytes({9, 10, 11}));
}

TEST(CropTest, WholeImageIsACopy) {
    const auto src = ramp(3, 4, 2);
    std::vector<uint8_t> dst(src.size());
    crop(src.data(), 3, 4, 2, 0, 0, 4, 3, dst.data());
    EXPECT_EQ(dst, src);
}

TEST(CropTest, RejectsBoxesOutsideTheImage) {
    const auto src = ramp(3, 4);
    std::vector<uint8_t> dst(12);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { crop(src.data(), 3, 4, 1, 0, 0, 5, 3, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { crop(src.data(), 3, 4, 1, -1, 0, 4, 3, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { crop(src.data(), 3, 4, 1, 0, 0, 4, 4, dst.data()); });
}

TEST(CropTest, RejectsEmptyBoxes) {
    const auto src = ramp(3, 4);
    std::vector<uint8_t> dst(12);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { crop(src.data(), 3, 4, 1, 2, 0, 2, 3, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { crop(src.data(), 3, 4, 1, 3, 1, 1, 2, dst.data()); });
}

// ------------------------------------------------------------ warpByGrid ---

/** A `[2, gh, gw]` identity field: x runs -1..1 across, y runs -1..1 down. */
std::vector<float> identityGrid(int32_t gh, int32_t gw) {
    std::vector<float> grid(static_cast<size_t>(2) * gh * gw);
    for (int32_t y = 0; y < gh; ++y) {
        for (int32_t x = 0; x < gw; ++x) {
            grid[y * gw + x] = gw > 1 ? -1.0f + 2.0f * static_cast<float>(x) / static_cast<float>(gw - 1) : 0.0f;
            grid[gh * gw + y * gw + x] = gh > 1 ? -1.0f + 2.0f * static_cast<float>(y) / static_cast<float>(gh - 1) : 0.0f;
        }
    }
    return grid;
}

TEST(WarpByGridTest, IdentityGridCopiesTheImage) {
    const auto src = ramp(6, 9, 3);
    const auto grid = identityGrid(2, 2);
    std::vector<uint8_t> dst(src.size());
    WarpOptions options;
    options.interpolation = Interpolation::Nearest;
    warpByGrid(src.data(), 6, 9, 3, grid.data(), 2, 2, 6, 9, options, dst.data());
    EXPECT_EQ(dst, src);
}

TEST(WarpByGridTest, LinearAndCubicSamplingAreExactOnPixelCentres) {
    const auto src = ramp(5, 7, 1);
    const auto grid = identityGrid(3, 4);
    for (const auto interpolation : {Interpolation::Linear, Interpolation::Cubic}) {
        std::vector<uint8_t> dst(src.size());
        WarpOptions options;
        options.interpolation = interpolation;
        warpByGrid(src.data(), 5, 7, 1, grid.data(), 3, 4, 5, 7, options, dst.data());
        // Fixed-point remap coordinates may land a hair off a centre.
        expectClose(dst, src, 1);
    }
}

TEST(WarpByGridTest, MirroredGridFlipsHorizontally) {
    const auto src = ramp(2, 4); // [[0,1,2,3],[4,5,6,7]]
    auto grid = identityGrid(2, 2);
    // Negate the x channel: the left edge samples the right edge and back.
    for (size_t i = 0; i < 4; ++i) {
        grid[i] = -grid[i];
    }
    std::vector<uint8_t> dst(src.size());
    WarpOptions options;
    options.interpolation = Interpolation::Nearest;
    warpByGrid(src.data(), 2, 4, 1, grid.data(), 2, 2, 2, 4, options, dst.data());
    EXPECT_EQ(dst, bytes({3, 2, 1, 0, 7, 6, 5, 4}));
}

TEST(WarpByGridTest, UpsamplesACoarseGridLinearly) {
    // A 2x2 grid whose x channel is constant 0 samples the middle column
    // everywhere; upsampling to any size must keep that constant.
    const auto src = bytes({10, 20, 30, 40, 50, 60}); // 2x3
    std::vector<float> grid = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, 1.0f, 1.0f};
    std::vector<uint8_t> dst(6);
    WarpOptions options;
    options.interpolation = Interpolation::Nearest;
    warpByGrid(src.data(), 2, 3, 1, grid.data(), 2, 2, 2, 3, options, dst.data());
    EXPECT_EQ(dst, bytes({20, 20, 20, 50, 50, 50}));
}

TEST(WarpByGridTest, ResizesWhenTheDestinationDiffers) {
    // 2x2 source scaled to 3x3: the corners land on source corners and the
    // centre is the bilinear mean of all four.
    const auto src = bytes({0, 100, 200, 255});
    const auto grid = identityGrid(2, 2);
    std::vector<uint8_t> dst(9);
    warpByGrid(src.data(), 2, 2, 1, grid.data(), 2, 2, 3, 3, WarpOptions{}, dst.data());
    EXPECT_EQ(dst[0], 0);
    EXPECT_EQ(dst[2], 100);
    EXPECT_EQ(dst[6], 200);
    EXPECT_EQ(dst[8], 255);
    EXPECT_NEAR(dst[4], 139, 2); // (0 + 100 + 200 + 255) / 4
    EXPECT_NEAR(dst[1], 50, 2);
    EXPECT_NEAR(dst[3], 100, 2);
}

TEST(WarpByGridTest, ConstantBorderFillsSamplesOutsideTheSource) {
    const auto src = ramp(2, 3);
    // x = 3 is far right of the source for every destination pixel.
    std::vector<float> grid = {3.0f, 3.0f, 3.0f, 3.0f, -1.0f, -1.0f, 1.0f, 1.0f};
    std::vector<uint8_t> dst(6);
    WarpOptions options;
    options.borderMode = BorderMode::Constant;
    options.borderValue = 7;
    warpByGrid(src.data(), 2, 3, 1, grid.data(), 2, 2, 2, 3, options, dst.data());
    EXPECT_EQ(dst, bytes({7, 7, 7, 7, 7, 7}));
}

TEST(WarpByGridTest, ReplicateBorderRepeatsTheEdge) {
    const auto src = ramp(2, 3); // rows end in 2 and 5
    std::vector<float> grid = {3.0f, 3.0f, 3.0f, 3.0f, -1.0f, -1.0f, 1.0f, 1.0f};
    std::vector<uint8_t> dst(6);
    WarpOptions options;
    options.interpolation = Interpolation::Nearest;
    warpByGrid(src.data(), 2, 3, 1, grid.data(), 2, 2, 2, 3, options, dst.data());
    EXPECT_EQ(dst, bytes({2, 2, 2, 5, 5, 5}));
}

TEST(WarpByGridTest, RejectsMoreThanFourChannels) {
    const auto src = ramp(2, 2, 5);
    const auto grid = identityGrid(2, 2);
    std::vector<uint8_t> dst(src.size());
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { warpByGrid(src.data(), 2, 2, 5, grid.data(), 2, 2, 2, 2, WarpOptions{}, dst.data()); });
}

TEST(WarpByGridTest, RejectsEmptyDimensions) {
    const auto src = ramp(2, 2);
    const auto grid = identityGrid(2, 2);
    std::vector<uint8_t> dst(4);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { warpByGrid(src.data(), 2, 2, 1, grid.data(), 0, 2, 2, 2, WarpOptions{}, dst.data()); });
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { warpByGrid(src.data(), 2, 2, 1, grid.data(), 2, 2, 2, 0, WarpOptions{}, dst.data()); });
}

// --------------------------------------------------------- padCornerMean ---

TEST(PadCornerMeanTest, FillsTheNewColumnsWithTheCornerMean) {
    // 4x6 strip, one-pixel corner patches: 8, 12, 8, 12 -> mean 10. The
    // interior is 50 so a mean over the whole strip would be visibly wrong.
    std::vector<uint8_t> strip(4 * 6, 50);
    strip[0] = 8;
    strip[5] = 12;
    strip[3 * 6] = 8;
    strip[3 * 6 + 5] = 12;
    std::vector<uint8_t> dst(4 * 10);
    padCornerMean(strip.data(), 4, 6, 1, 10, dst.data());
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 10; ++x) {
            const uint8_t expected = x < 6 ? strip[y * 6 + x] : 10;
            EXPECT_EQ(dst[y * 10 + x], expected) << "at (" << x << ", " << y << ")";
        }
    }
}

TEST(PadCornerMeanTest, AveragesPerChannel) {
    std::vector<uint8_t> strip(2 * 2 * 3, 0);
    for (int p = 0; p < 4; ++p) {
        strip[p * 3] = 10;
        strip[p * 3 + 1] = 20;
        strip[p * 3 + 2] = 30;
    }
    std::vector<uint8_t> dst(2 * 3 * 3);
    padCornerMean(strip.data(), 2, 2, 3, 3, dst.data());
    for (int y = 0; y < 2; ++y) {
        const size_t pad = (y * 3 + 2) * 3;
        EXPECT_EQ(dst[pad], 10);
        EXPECT_EQ(dst[pad + 1], 20);
        EXPECT_EQ(dst[pad + 2], 30);
    }
}

TEST(PadCornerMeanTest, RoundsTheMean) {
    // Corners 1, 2, 2, 2 -> 1.75 -> 2.
    std::vector<uint8_t> strip = {1, 2, 2, 2};
    std::vector<uint8_t> dst(2 * 3);
    padCornerMean(strip.data(), 2, 2, 1, 3, dst.data());
    EXPECT_EQ(dst[2], 2);
    EXPECT_EQ(dst[5], 2);
}

TEST(PadCornerMeanTest, EqualWidthIsACopy) {
    const auto strip = ramp(3, 5, 4);
    std::vector<uint8_t> dst(strip.size());
    padCornerMean(strip.data(), 3, 5, 4, 5, dst.data());
    EXPECT_EQ(dst, strip);
}

TEST(PadCornerMeanTest, RejectsANarrowerTarget) {
    const auto strip = ramp(3, 5);
    std::vector<uint8_t> dst(15);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { padCornerMean(strip.data(), 3, 5, 1, 4, dst.data()); });
}

TEST(PadCornerMeanTest, RejectsMoreThanFourChannels) {
    const auto strip = ramp(2, 2, 5);
    std::vector<uint8_t> dst(2 * 3 * 5);
    expectThrowsCode(ErrorCode::InvalidArgument, [&] { padCornerMean(strip.data(), 2, 2, 5, 3, dst.data()); });
}

} // namespace
} // namespace rnocr::tests
